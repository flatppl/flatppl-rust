module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<f32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<0.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<2.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.divide %3, %2 : tensor<f32>
    %5 = stablehlo.constant dense<0.0> : tensor<f32>
    %6 = stablehlo.constant dense<1.0> : tensor<f32>
    %7 = stablehlo.compare LT, %4, %6 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = stablehlo.add %4, %6 : tensor<f32>
    %9 = stablehlo.select %7, %8, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %10 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %11 = stablehlo.subtract %9, %10 : tensor<f32>
    %12 = stablehlo.constant dense<9.0> : tensor<f32>
    %13 = stablehlo.multiply %12, %11 : tensor<f32>
    %14 = stablehlo.sqrt %13 : tensor<f32>
    %15 = stablehlo.divide %6, %14 : tensor<f32>
    %16, %17 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %18 = stablehlo.constant dense<9> : tensor<128xui32>
    %19 = stablehlo.shift_right_logical %17, %18 : tensor<128xui32>
    %20 = stablehlo.convert %19 : (tensor<128xui32>) -> tensor<128xf32>
    %21 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<128xf32>
    %23 = stablehlo.constant dense<2.0> : tensor<128xf32>
    %24 = stablehlo.constant dense<1.0> : tensor<128xf32>
    %25 = stablehlo.multiply %22, %23 : tensor<128xf32>
    %26 = stablehlo.subtract %25, %24 : tensor<128xf32>
    %27 = chlo.erf_inv %26 : tensor<128xf32> -> tensor<128xf32>
    %28 = stablehlo.constant dense<1.4142135> : tensor<128xf32>
    %29 = stablehlo.multiply %27, %28 : tensor<128xf32>
    %30, %31 = stablehlo.rng_bit_generator %16, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128xui32>)
    %32 = stablehlo.constant dense<9> : tensor<128xui32>
    %33 = stablehlo.shift_right_logical %31, %32 : tensor<128xui32>
    %34 = stablehlo.convert %33 : (tensor<128xui32>) -> tensor<128xf32>
    %35 = stablehlo.constant dense<1.1920929E-7> : tensor<128xf32>
    %36 = stablehlo.multiply %34, %35 : tensor<128xf32>
    %37 = stablehlo.constant dense<0> : tensor<i32>
    %38 = stablehlo.constant dense<false> : tensor<i1>
    %39 = stablehlo.constant dense<0.0> : tensor<f32>
    %43:3 = stablehlo.while(%40 = %37, %41 = %38, %42 = %39) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %44 = stablehlo.constant dense<128> : tensor<i32>
      %45 = stablehlo.compare LT, %40, %44, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %46 = stablehlo.not %41 : tensor<i1>
      %47 = stablehlo.and %46, %45 : tensor<i1>
      stablehlo.return %47 : tensor<i1>
    } do {
      %48 = stablehlo.dynamic_slice %29, %40, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %49 = stablehlo.reshape %48 : (tensor<1xf32>) -> tensor<f32>
      %50 = stablehlo.dynamic_slice %36, %40, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %51 = stablehlo.reshape %50 : (tensor<1xf32>) -> tensor<f32>
      %52 = stablehlo.multiply %15, %49 : tensor<f32>
      %53 = stablehlo.add %6, %52 : tensor<f32>
      %54 = stablehlo.multiply %53, %53 : tensor<f32>
      %55 = stablehlo.multiply %54, %53 : tensor<f32>
      %56 = stablehlo.multiply %11, %55 : tensor<f32>
      %57 = stablehlo.constant dense<0.5> : tensor<f32>
      %58 = stablehlo.multiply %49, %49 : tensor<f32>
      %59 = stablehlo.multiply %57, %58 : tensor<f32>
      %60 = stablehlo.multiply %11, %55 : tensor<f32>
      %61 = stablehlo.negate %60 : tensor<f32>
      %62 = stablehlo.log %55 : tensor<f32>
      %63 = stablehlo.multiply %11, %62 : tensor<f32>
      %64 = stablehlo.add %59, %11 : tensor<f32>
      %65 = stablehlo.add %64, %61 : tensor<f32>
      %66 = stablehlo.add %65, %63 : tensor<f32>
      %67 = stablehlo.log %51 : tensor<f32>
      %68 = stablehlo.compare LT, %67, %66 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %69 = stablehlo.compare GT, %55, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %70 = stablehlo.and %68, %69 : tensor<i1>
      %71 = stablehlo.constant dense<1> : tensor<i32>
      %72 = stablehlo.add %40, %71 : tensor<i32>
      stablehlo.return %72, %70, %56 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %73, %74 = stablehlo.rng_bit_generator %30, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %75 = stablehlo.constant dense<9> : tensor<ui32>
    %76 = stablehlo.shift_right_logical %74, %75 : tensor<ui32>
    %77 = stablehlo.convert %76 : (tensor<ui32>) -> tensor<f32>
    %78 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %79 = stablehlo.multiply %77, %78 : tensor<f32>
    %80 = stablehlo.divide %6, %4 : tensor<f32>
    %81 = stablehlo.power %79, %80 : tensor<f32>
    %82 = stablehlo.select %7, %81, %6 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %83 = stablehlo.multiply %43#2, %82 : tensor<f32>
    %84 = stablehlo.divide %83, %3 : tensor<f32>
    %85 = stablehlo.power %84, %4 : tensor<f32>
    %86 = stablehlo.constant dense<0.0> : tensor<f32>
    %87, %88 = stablehlo.rng_bit_generator %73, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui32>)
    %89 = stablehlo.constant dense<9> : tensor<ui32>
    %90 = stablehlo.shift_right_logical %88, %89 : tensor<ui32>
    %91 = stablehlo.convert %90 : (tensor<ui32>) -> tensor<f32>
    %92 = stablehlo.constant dense<1.1920929E-7> : tensor<f32>
    %93 = stablehlo.multiply %91, %92 : tensor<f32>
    %94 = stablehlo.constant dense<0.5> : tensor<f32>
    %95 = stablehlo.subtract %93, %94 : tensor<f32>
    %96 = stablehlo.compare GE, %95, %86 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %97 = stablehlo.constant dense<1.0> : tensor<f32>
    %98 = stablehlo.constant dense<-1.0> : tensor<f32>
    %99 = stablehlo.select %96, %97, %98 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %100 = stablehlo.multiply %1, %99 : tensor<f32>
    %101 = stablehlo.multiply %100, %85 : tensor<f32>
    %102 = stablehlo.add %0, %101 : tensor<f32>
    return %102, %87 : tensor<f32>, tensor<2xui64>
  }
}

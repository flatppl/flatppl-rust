module {
  func.func @sample() -> tensor<f32> {
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
    %16 = stablehlo.constant dense<128> : tensor<1xi64>
    %17 = stablehlo.rng %5, %6, %16, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %18 = stablehlo.constant dense<128> : tensor<1xi64>
    %19 = stablehlo.rng %5, %6, %18, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %20 = stablehlo.constant dense<0> : tensor<i32>
    %21 = stablehlo.constant dense<false> : tensor<i1>
    %22 = stablehlo.constant dense<0.0> : tensor<f32>
    %26:3 = stablehlo.while(%23 = %20, %24 = %21, %25 = %22) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %27 = stablehlo.constant dense<128> : tensor<i32>
      %28 = stablehlo.compare LT, %23, %27, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %29 = stablehlo.not %24 : tensor<i1>
      %30 = stablehlo.and %29, %28 : tensor<i1>
      stablehlo.return %30 : tensor<i1>
    } do {
      %31 = stablehlo.dynamic_slice %17, %23, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
      %33 = stablehlo.dynamic_slice %19, %23, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %34 = stablehlo.reshape %33 : (tensor<1xf32>) -> tensor<f32>
      %35 = stablehlo.multiply %15, %32 : tensor<f32>
      %36 = stablehlo.add %6, %35 : tensor<f32>
      %37 = stablehlo.multiply %36, %36 : tensor<f32>
      %38 = stablehlo.multiply %37, %36 : tensor<f32>
      %39 = stablehlo.multiply %11, %38 : tensor<f32>
      %40 = stablehlo.constant dense<0.5> : tensor<f32>
      %41 = stablehlo.multiply %32, %32 : tensor<f32>
      %42 = stablehlo.multiply %40, %41 : tensor<f32>
      %43 = stablehlo.multiply %11, %38 : tensor<f32>
      %44 = stablehlo.negate %43 : tensor<f32>
      %45 = stablehlo.log %38 : tensor<f32>
      %46 = stablehlo.multiply %11, %45 : tensor<f32>
      %47 = stablehlo.add %42, %11 : tensor<f32>
      %48 = stablehlo.add %47, %44 : tensor<f32>
      %49 = stablehlo.add %48, %46 : tensor<f32>
      %50 = stablehlo.log %34 : tensor<f32>
      %51 = stablehlo.compare LT, %50, %49 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %52 = stablehlo.compare GT, %38, %5 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %53 = stablehlo.and %51, %52 : tensor<i1>
      %54 = stablehlo.constant dense<1> : tensor<i32>
      %55 = stablehlo.add %23, %54 : tensor<i32>
      stablehlo.return %55, %53, %39 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %56 = stablehlo.constant dense<> : tensor<0xi64>
    %57 = stablehlo.rng %5, %6, %56, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %58 = stablehlo.divide %6, %4 : tensor<f32>
    %59 = stablehlo.power %57, %58 : tensor<f32>
    %60 = stablehlo.select %7, %59, %6 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %61 = stablehlo.multiply %26#2, %60 : tensor<f32>
    %62 = stablehlo.divide %61, %3 : tensor<f32>
    %63 = stablehlo.power %62, %4 : tensor<f32>
    %64 = stablehlo.constant dense<0.0> : tensor<f32>
    %65 = stablehlo.constant dense<> : tensor<0xi64>
    %66 = stablehlo.rng %64, %3, %65, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %67 = stablehlo.constant dense<0.5> : tensor<f32>
    %68 = stablehlo.subtract %66, %67 : tensor<f32>
    %69 = stablehlo.compare GE, %68, %64 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %70 = stablehlo.constant dense<1.0> : tensor<f32>
    %71 = stablehlo.constant dense<-1.0> : tensor<f32>
    %72 = stablehlo.select %69, %70, %71 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %73 = stablehlo.multiply %1, %72 : tensor<f32>
    %74 = stablehlo.multiply %73, %63 : tensor<f32>
    %75 = stablehlo.add %0, %74 : tensor<f32>
    return %75 : tensor<f32>
  }
}
